import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import os
from model_definitions import EEGTransformer 


def load_model(model_type, model_path, num_classes):
    if model_type == "transformer":
        model = EEGTransformer(
            input_dim=20,
            model_dim=128,
            num_heads=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            num_classes=num_classes
        )
    elif model_type == "cnn":
        model = SpectrogramCNN(
            num_classes=num_classes,
            in_channels=1,
            base_filters=16
        )
    else:
        raise ValueError("invalid model_type. must be 'transformer' or 'cnn'.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"could not find model file: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(model, input_file):
    df = pd.read_parquet(input_file)
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    data_tensor = data_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(data_tensor)  # => (1, num_classes)
        predicted_idx = torch.argmax(outputs, dim=1).item()  # => integer index
    return predicted_idx


def main():
    model_type = "transformer"
    model_path = "transformer5.pth"
    input_file = "G:/my drive/fau/egn4952c_spring_2025/data/labeled_training_eegs/5423338_0_0_LPD.parquet"
    # input_file = "G:/my drive/fau/egn4952c_spring_2025/data/labeled_training_eegs/9859330_8_16_GPD.parquet"
    # input_file = "G:/my drive/fau/egn4952c_spring_2025/data/labeled_training_eegs/14623517_1_6_Seizure.parquet"
    # input_file = "G:/my drive/fau/egn4952c_spring_2025/data/labeled_training_eegs/17295749_5_14_GRDA.parquet"
    # input_file = "G:/my drive/fau/egn4952c_spring_2025/data/labeled_training_eegs/568657_2_12_Other.parquet"
    num_classes = 6
    index_to_label = {
        0: "Seizure",
        1: "LPD",
        2: "GPD",
        3: "LRDA",
        4: "GRDA",
        5: "Other"
    }
    model = load_model(model_type, model_path, num_classes)
    predicted_idx = predict(model, input_file)
    predicted_label = index_to_label.get(predicted_idx, "unknown")
    print(f"predicted class index: {predicted_idx}")
    print(f"predicted label: {predicted_label}")


if __name__ == "__main__":
    main()
