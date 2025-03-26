import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dl.transformer.train import config, EEGDataset, EEGTransformer

def train():
  dataset = EEGDataset(
    config["data_folder"], 
    config["label_index"]
  )
  loader = DataLoader(
    dataset, 
    batch_size = config["batch_size"], 
    shuffle = True
  )
  m = EEGTransformer(
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
  o = optim.Adam(m.parameters(), lr=config["learning_rate"])
  c = nn.CrossEntropyLoss()
  m.train()
  for num_epochs in range(config["num_epochs"]):
    L = 0.0
    for x, y in loader:
      o.zero_grad()
      z = m(x)
      l = c(z, y)
      l.backward()
      o.step()
      L += l.item()
    print(num_epochs + 1, L / len(loader))

if __name__ == "__main__":
  train()
