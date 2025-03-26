import torch
from torch.nn import DataParallel
from . import config, EEGTransformer

def infer(inputs, model_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = EEGTransformer(
    input_dim = config["input_dim"],
    model_dim = config["model_dim"],
    num_heads = config["num_heads"],
    num_layers = config["num_layers"],
    dim_feedforward = config["dim_feedforward"],
    dropout = config["dropout"],
    num_classes = config["num_classes"],
    max_len = config["max_len"],
    use_learnable_pos_emb = config["use_learnable_pos_emb"],
    use_cls_token = config["use_cls_token"],
    pooling = config["pooling"]
  )
  if torch.cuda.device_count() > 1:
    model = DataParallel(model)
  model.to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
  with torch.no_grad():
    inputs = inputs.to(device)
    return model(inputs)
