import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    """
    A CNN for spectrogram classification.

    Assumes each input is shaped as [batch_size, time, freq_bins] 
    (e.g., [batch_size, 100, 401]), and we treat this as 
    a single-channel image of size (time x freq_bins).
    """
    def __init__(self, num_classes=6):
        super(SpectrogramCNN, self).__init__()
        
        # Convolutional layers (you can adjust the number of filters, kernel sizes, etc.)
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # halves each dimension
            
            # Conv Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # further reduces height and width

            # Conv Block 3 (optional, add more if needed)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # further reduces height and width
        )

        # After three pooling layers, each dimension is reduced by a factor of 2^3 = 8.
        # If your input is (time x freq_bins) = (T x 401), after the conv block it will be:
        #   [batch_size, 64, T/8, 401/8]
        # We'll need to compute the flattened dimension for the final linear layers. 
        # Because time dimension can vary, we do it dynamically in forward().
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=64 * 1 * 1, out_features=128),  # placeholder; we’ll set correct size in forward()
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        """
        Expects x of shape [batch_size, time, freq_bins].
        We'll insert a channel dimension => shape [batch_size, 1, time, freq_bins].
        """
        # 1. Add channel dimension
        x = x.unsqueeze(1)  # shape: [batch_size, 1, time, freq_bins]

        # 2. Pass through convolutional layers
        x = self.conv_layers(x)
        
        # x shape is now [batch_size, 64, new_time, new_freq]
        # We need to flatten for the FC layers. But first, we’ll figure out the new shape:
        bsz, c, h, w = x.shape  # c = 64, h/w after pooling
        
        # 3. Flatten
        x = x.view(bsz, -1)  # shape: [batch_size, 64 * h * w]
        
        # 4. If the linear layers have a mismatch, we handle it dynamically:
        #    We'll replace the first Linear layer if needed.
        if self.fc_layers[0].in_features != c * h * w:
            self.fc_layers[0] = nn.Linear(c * h * w, 128)
        
        # 5. Pass through fully connected layers
        x = self.fc_layers(x)  # shape: [batch_size, num_classes]

        return x


'''
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class SpecCNN(nn.Module):
    def __init__( self, num_classes=6, in_channels=1,  base_filters=16 ):
        super(SpecCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(base_filters)

        self.conv2 = nn.Conv1d(base_filters, base_filters*2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(base_filters*2)

        self.conv3 = nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(base_filters*4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(base_filters*4, num_classes)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)   
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)  

        x = self.conv2(x)   
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)  

        x = self.conv3(x)  
        x = self.bn3(x)
        x = F.relu(x)

        x = self.pool(x)   
        x = x.squeeze(-1)  
        x = self.fc(x)     

        return x
'''
