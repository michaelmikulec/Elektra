import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

DATA_FOLDER = "/path/to/eeg/parquet_files"
LABEL_INDEX = {
    "Seizure": 0,
    "LRDA": 1,
    "GRDA": 2,
    "LPD": 3,
    "GPD": 4,
    "Other": 5
}
INPUT_DIM = 20            # Example: 20 EEG channels
MODEL_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
NUM_CLASSES = len(LABEL_INDEX)
MAX_LEN = 12000  # Based on your expected maximum sequence length
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
VAL_SPLIT = 0.2  # Fraction of data to use for validation
SEED = 42        # For reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        data, labels = batch
        data = data.to(device)    # shape: [batch_size, seq_len, input_dim]
        labels = labels.to(device)  # shape: [batch_size]

        # Forward pass
        outputs = model(data)  # shape: [batch_size, num_classes]
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * data.size(0)
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
    #    This expects you have an EEGDataset class that extracts labels from filenames
    dataset = EEGDataset(
        data_folder=DATA_FOLDER,
        labels_list=list(LABEL_INDEX.keys())  # or provide a label_index in your custom dataset
    )

    # 3. Split into train/validation
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 5. Initialize the EEGTransformer model
    model = EEGTransformer(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(device)

    # 6. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Optionally, you could use a scheduler, for example:
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_path = "best_eeg_transformer.pth"

    # 7. Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        # Optionally, step the scheduler based on validation accuracy or loss
        # scheduler.step(val_acc) or scheduler.step(val_loss)

        # Print stats
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
