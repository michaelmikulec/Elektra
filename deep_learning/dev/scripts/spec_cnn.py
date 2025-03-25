import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SpectrogramCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=64 * 1 * 1, out_features=128),  
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv_layers(x)
        bsz, c, h, w = x.shape 
        x = x.view(bsz, -1) 
        if self.fc_layers[0].in_features != c * h * w:
            self.fc_layers[0] = nn.Linear(c * h * w, 128)
        x = self.fc_layers(x) 
        return x

