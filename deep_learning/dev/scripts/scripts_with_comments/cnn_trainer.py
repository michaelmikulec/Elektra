import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np

# Make sure to import or define SpecDataset and SpectrogramCNN from your files:
# from spec_dataset import SpecDataset
# from cnn_model import SpectrogramCNN

# For this example, let's assume these classes are in the same file or have been defined above:
# class SpecDataset(Dataset): ...
# class SpectrogramCNN(nn.Module): ...

#####################
# Hyperparameters
#####################
DATA_FOLDER = "/path/to/parquet_spectrograms"
LABEL_INDEX = {
    "Seizure": 0,
    "LRDA": 1,
    "GRDA": 2,
    "LPD": 3,
    "GPD": 4,
    "Other": 5
}
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
VAL_SPLIT = 0.2   # Fraction of data to use for validation

# Optional: set a random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    One epoch of training. Returns average loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        spectrograms, labels = batch
        spectrograms = spectrograms.to(device)  # [batch_size, time, 401]
        labels = labels.to(device)              # [batch_size]

        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * spectrograms.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """
    One epoch of validation. Returns average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            spectrograms, labels = batch
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            # Stats
            running_loss += loss.item() * spectrograms.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # 1. Device configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create the dataset
    dataset = SpecDataset(
        data_folder=DATA_FOLDER,
        label_index=LABEL_INDEX
    )

    # 3. Split into train and validation
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 5. Initialize the CNN model
    model = SpectrogramCNN(num_classes=len(LABEL_INDEX)).to(device)

    # 6. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Optionally, you could use a scheduler, for example:
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_path = "best_spectrogram_cnn.pth"

    # 7. Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        # Optionally step the scheduler based on validation accuracy
        # scheduler.step(val_acc)

        # Print epoch stats
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 8. Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")

    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model is saved to: {best_model_path}")

if __name__ == "__main__":
    main()
