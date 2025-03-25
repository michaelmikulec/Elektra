# main.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# 1) Import your dataset classes, model classes, etc.
#    Adjust these paths/names to match your actual file structure.
# from eeg_dataset import EEGDataset
# from spec_dataset import SpecDataset
# from eeg_transformer import EEGTransformer
# from spectrogram_cnn import SpectrogramCNN

# For illustration, we define stubs here. Remove or comment out these stubs and uncomment real imports above.
class EEGDataset:
    def __init__(self, data_folder, label_index):
        # Stub: Provide your own logic to load EEG data
        self.data_folder = data_folder
        self.label_index = label_index
        self.data_files = []  # collect .parquet files
        # ...
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        # return (data_tensor, label_tensor)
        pass

class SpecDataset:
    def __init__(self, data_folder, label_index):
        # Stub: Provide your own logic to load spectrogram data
        self.data_folder = data_folder
        self.label_index = label_index
        self.data_files = []  # collect .parquet files
        # ...
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        # return (data_tensor, label_tensor)
        pass

class EEGTransformer(nn.Module):
    def __init__(
        self,
        input_dim=20,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=6
    ):
        super().__init__()
        # Stub
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x.mean(dim=1))  # naive forward

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Stub
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 1 * 1, num_classes)  # placeholder
    def forward(self, x):
        # x shape: [batch_size, time, freq_bins] => add channel dimension
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv(x)))
        # Flatten
        b, c, h, w = x.shape
        x = x.view(b, -1)
        return self.fc(x)

# 2) Shared label dictionary (for both EEG and Spectrogram data if they share the same labels)
LABEL_INDEX = {
    "Seizure": 0,
    "LRDA": 1,
    "GRDA": 2,
    "LPD": 3,
    "GPD": 4,
    "Other": 5
}

# Paths to labeled EEG and Spectrogram data
EEG_DATA_FOLDER = "G:/My Drive/fau/egn4952c_spring_2025/data/labeled_eeg_training_data/"
SPEC_DATA_FOLDER = "G:/My Drive/fau/egn4952c_spring_2025/data/labeled_spec_training_data/"

# Model checkpoints
BEST_EEG_TRANSFORMER_PATH = "best_eeg_transformer.pth"
BEST_SPECTROGRAM_CNN_PATH = "best_spectrogram_cnn.pth"

# Hyperparameters
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 5
VAL_SPLIT = 0.2

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_eeg_transformer(model, dataset, device, epochs=EPOCHS):
    """
    A simple training loop for EEGTransformer. 
    This can be replaced by your more elaborate training script.
    """

    # 1. Split dataset
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss_accum += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss_accum / val_total
        val_acc = val_correct / val_total

        print(f"[EEGTransformer] Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_EEG_TRANSFORMER_PATH)
            print(f"  >>> Best EEGTransformer saved with val_acc={val_acc:.4f}")

def train_spectrogram_cnn(model, dataset, device, epochs=EPOCHS):
    """
    A simple training loop for SpectrogramCNN.
    """
    # 1. Split dataset
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss_accum += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss_accum / val_total
        val_acc = val_correct / val_total

        print(f"[SpectrogramCNN] Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_SPECTROGRAM_CNN_PATH)
            print(f"  >>> Best SpectrogramCNN saved with val_acc={val_acc:.4f}")


def main():
    """
    - Loads labeled EEG and spectrogram data.
    - Attempts to load existing transformer and CNN models, or trains new ones if not found.
    - Saves both models to files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load the datasets
    eeg_dataset = EEGDataset(
        data_folder=EEG_DATA_FOLDER,
        label_index=LABEL_INDEX  # or pass in a list of labels; depends on your dataset implementation
    )
    spec_dataset = SpecDataset(
        data_folder=SPEC_DATA_FOLDER,
        label_index=LABEL_INDEX
    )

    # 2) Create model instances
    # EEG Transformer
    eeg_transformer = EEGTransformer(
        input_dim=20,       # match your data dimension
        model_dim=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=len(LABEL_INDEX)
    ).to(device)

    # Spectrogram CNN
    spectrogram_cnn = SpectrogramCNN(
        num_classes=len(LABEL_INDEX)
    ).to(device)

    # 3) Check if pretrained models exist. If they do, load them; otherwise train from scratch.
    # ---- EEG Transformer ----
    if os.path.exists(BEST_EEG_TRANSFORMER_PATH):
        print(f"Found existing EEG Transformer model at {BEST_EEG_TRANSFORMER_PATH}. Loading weights...")
        state_dict = torch.load(BEST_EEG_TRANSFORMER_PATH, map_location=device)
        eeg_transformer.load_state_dict(state_dict)
    else:
        # Train a new EEG Transformer
        print(f"No existing EEG Transformer model found. Training a new one...")
        train_eeg_transformer(eeg_transformer, eeg_dataset, device, epochs=EPOCHS)

    # ---- Spectrogram CNN ----
    if os.path.exists(BEST_SPECTROGRAM_CNN_PATH):
        print(f"Found existing Spectrogram CNN model at {BEST_SPECTROGRAM_CNN_PATH}. Loading weights...")
        state_dict = torch.load(BEST_SPECTROGRAM_CNN_PATH, map_location=device)
        spectrogram_cnn.load_state_dict(state_dict)
    else:
        # Train a new CNN
        print(f"No existing Spectrogram CNN model found. Training a new one...")
        train_spectrogram_cnn(spectrogram_cnn, spec_dataset, device, epochs=EPOCHS)

    print("Done. Models are ready.")

if __name__ == "__main__":
    main()
