import torch
import torch.nn as nn

class EEGCNN(nn.Module):
  def __init__(
    self,
    input_dim=20,
    num_filters=32,
    num_layers=2,
    kernel_size=5,
    dropout=0.1,
    num_classes=6
  ):
    super().__init__()
    self.input_dim = input_dim
    self.num_filters = num_filters
    layers = []
    in_channels = input_dim
    for _ in range(num_layers):
      conv = nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2)
      bn = nn.BatchNorm1d(num_filters)
      relu = nn.ReLU()
      layers.append(conv)
      layers.append(bn)
      layers.append(relu)
      in_channels = num_filters
    self.conv_stack = nn.Sequential(*layers)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(num_filters * 10000, num_classes)
  
  def forward(self, x):
    x = x.transpose(1, 2)
    x = self.conv_stack(x)
    x = x.reshape(x.size(0), -1)
    x = self.dropout(x)
    x = self.fc(x)
    return x
