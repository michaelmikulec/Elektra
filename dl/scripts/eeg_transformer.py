import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import pandas as pd

class EEGDataset(Dataset):
  def __init__(self,data_folder,label_index):
    self.data_folder=data_folder
    self.data_files=sorted([os.path.join(data_folder,f) for f in os.listdir(data_folder) if f.endswith(".parquet")])
    self.label_index=label_index
    self.int_labels=[]
    self.str_labels=[]
    for fp in self.data_files:
      int_label,str_label=self.extract_labels(fp)
      if str_label not in label_index:
        raise ValueError(f"File '{fp}' has string label '{str_label}', but it's not in {list(label_index.keys())}")
      self.int_labels.append(int_label)
      self.str_labels.append(str_label)

  def extract_labels(self,filepath):
    base=os.path.basename(filepath)
    parts=base.split("_")
    int_label=int(parts[0])
    str_label=parts[1]
    return int_label,str_label

  def __len__(self):
    return len(self.data_files)

  def __getitem__(self,idx):
    fp=self.data_files[idx]
    df=pd.read_parquet(fp)
    data_sample=torch.tensor(df.values,dtype=torch.float32)
    mapped_label=self.label_index[self.str_labels[idx]]
    label_tensor=torch.tensor(mapped_label,dtype=torch.long)
    return data_sample,label_tensor

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=12000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.pe.size(1)}. "
                "Increase max_len or downsample the data."
            )
        x = x + self.pe[:, :seq_len, :]
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=12000):
        super().__init__()
        
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.pe.size(1)}. "
                "Increase max_len or downsample the data."
            )
        x = x + self.pe[:, :seq_len, :]
        return x

class EEGTransformer(nn.Module):
    def __init__(
        self,
        input_dim=20,
        model_dim=128,
        num_heads=4,  
        num_layers=2, 
        dim_feedforward=256,  
        dropout=0.1,
        num_classes=6,
        max_len=12000,
        use_learnable_pos_emb=True,
        use_cls_token=True,
        pooling="cls"
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.pooling = pooling
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.input_embed = nn.Linear(input_dim, model_dim)
        if use_learnable_pos_emb:
            self.pos_encoding = LearnablePositionalEncoding(model_dim, max_len=max_len)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(model_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),           
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02) if self.use_cls_token else None

    def forward(self, x):
        x = self.input_embed(x)  
        if self.use_cls_token:
            bsz = x.size(0)
            cls_tokens = self.cls_token.repeat(bsz, 1, 1)  
            x = torch.cat([cls_tokens, x], dim=1)          
        x = self.pos_encoding(x)  
        x = self.transformer_encoder(x)  
        if self.use_cls_token and self.pooling == "cls":
            x = x[:, 0, :]  
        else:
            x = x.mean(dim=1)  
        logits = self.classifier(x)  
        return logits

if __name__ == "__main__":
    fake_data = torch.randn(8, 1000, 20)
    model = EEGTransformer(
        input_dim=20,
        model_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=6,
        max_len=12000,
        use_learnable_pos_emb=True,
        use_cls_token=True,
        pooling="cls"
    )
    outputs = model(fake_data)
    print(outputs.shape)  
