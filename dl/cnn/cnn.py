import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
  def __init__(self, input_shape=(5, 400), num_classes=6):
    super().__init__()
    self.conv1   = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
    self.conv2   = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
    self.pool    = nn.AdaptiveAvgPool2d((2, 2))
    self.dropout = nn.Dropout(0.3)
    self.fc1     = nn.Linear(64 * 2 * 2, 128)
    self.fc2     = nn.Linear(128, num_classes)

  def forward(self, x):
    x = x.unsqueeze(1)              
    x = F.relu(self.conv1(x))       
    x = self.pool(F.relu(self.conv2(x)))
    x = self.dropout(x)
    x = torch.flatten(x, 1)         
    x = F.relu(self.fc1(x))         
    return self.fc2(x)              
