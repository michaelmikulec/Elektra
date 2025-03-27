import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from . import config, EEGDataset, EEGTransformer

def train():
  dataset = EEGDataset(config["training_data"], config["label_index"])
  loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
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
  optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
  criterion = nn.CrossEntropyLoss()
  model.train()
  for epoch in range(config["num_epochs"]):
    epoch_loss = 0.0
    for inputs, targets in loader:
      inputs = inputs.to(device)
      targets = targets.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss_value = criterion(outputs, targets)
      loss_value.backward()
      optimizer.step()
      epoch_loss += loss_value.item()
    print(epoch + 1, epoch_loss / len(loader))
  torch.save(model.state_dict(), config["model_save_path"])
