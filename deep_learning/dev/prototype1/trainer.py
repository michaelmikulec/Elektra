import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import os
import time
import pandas as pd
from tqdm import tqdm
from model_definitions import EEGTransformer, SpectrogramCNN 
from dataset_definitions import EEGDataset


def train_model(model, train_loader, criterion, optimizer, epochs=1, save_path="model.pth"):
    model.train()

    epoch_progress = tqdm(range(epochs), desc="Overall Training Progress")

    for epoch in epoch_progress:
        total_loss = 0.0
        correct = 0
        total = 0

        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for data, labels in batch_progress:
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            batch_progress.set_postfix({
                "loss": f"{total_loss / (total or 1):.4f}",
                "acc": f"{100.0 * correct / (total or 1):.2f}%"
            })

        epoch_progress.set_postfix({
            "epoch_loss": f"{total_loss / len(train_loader):.4f}",
            "epoch_acc": f"{100.0 * correct / total:.2f}%"
        })

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved as {save_path}")


def main(batch_size=1, 
         epochs=1, 
         lr=0.001, 
         num_workers=6, 
         num_threads=6,
         load=False, 
         model_path="model.pth", 
         model_type="transformer", 
         training_data_folder="data",
         labels_list=None):
    if labels_list is None:
        raise ValueError("You must provide labels_list to define the label order.")

    start = time.time()
    
    torch.set_num_threads(num_threads)

    print("Loading Training Data...")
    dataset = EEGDataset(training_data_folder, labels_list=labels_list)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    if model_type == "transformer":
        model = EEGTransformer(
            input_dim=dataset[0][0].shape[-1],
            model_dim=128,
            num_heads=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            num_classes=len(labels_list) 
        )
    elif model_type == "cnn":
        model = SpectrogramCNN(
            num_classes=len(labels_list),
            in_channels=1,
            base_filters=16
        )
    else:
        raise ValueError("Error: Invalid model type specified.")

    if load:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print("Setting up Loss and Optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting Training...")
    train_model(model, train_loader, criterion, optimizer, epochs=epochs, save_path=model_path)

    end = time.time()
    runtime = f"{((end - start)/3600):02.0f}:{((end - start)/60)%60:02.0f}:{(end - start)%60:02.0f}"
    print(f"runtime: {runtime}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    labels = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]

    main(
        batch_size=1,
        epochs=5,
        lr=0.001,
        num_workers=6,
        num_threads=16,
        load=False,
        model_path="transformer5.pth", 
        model_type="transformer",
        training_data_folder="G:/my drive/fau/egn4952c_spring_2025/ED1-PROJECT2024/labeled_training_data/labeled_training_eegs/subset/",
        labels_list=labels
    )
